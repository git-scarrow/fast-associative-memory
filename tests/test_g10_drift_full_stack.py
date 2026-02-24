"""
tests/test_g10_drift_full_stack.py — G10 Drift Benchmark with Full Stack.

Validates the non-destructive slot allocation property of FastAssociativeMemory
when the complete production stack is enabled:

  * ``use_bfloat16=True``          — quantized (bfloat16) prototype storage
  * :class:`~adapter.MetricAdapter` — learned metric-space projection
  * :class:`~nstp.NSTPController`   — post-retrieval lateral inhibition

Success criterion (G20 Continual-Learning Stress — Kill Condition):
  Backward Transfer (BWT) > -5 % on the G10 Drift Benchmark with
  full stack enabled.  A BWT ≤ -5 % fires the G20 kill condition.

All tests use synthetic low-dimensional embeddings so they run on CPU without
any external dataset (DINOv2 / ImageNet-R not required).
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter
from nstp import NSTPController

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_N_CLASSES  = 10
_DIM_IN     = 64   # backbone-like embedding dimension
_DIM_OUT    = 32   # adapter output / CAM key dimension
_N_TRAIN    = 50   # samples per class in Phase 1 (clean)
_N_DRIFT    = 50   # samples per class in Phase 2 (drift)
_BWT_THRESHOLD = -5.0   # kill condition: BWT must be > -5 %


def _make_clean_data(n_classes=_N_CLASSES, dim=_DIM_IN, n_per_class=_N_TRAIN, noise=0.05, seed=0):
    """Synthetic clean embeddings: small noise around orthogonal class centroids.

    Requires ``n_classes <= dim`` so that all centroids are distinct unit vectors
    (``torch.eye(n_classes, dim)`` rows 0…n_classes-1 are already orthogonal
    unit vectors when n_classes ≤ dim; with the defaults 10 ≤ 64 this always holds).

    Returns:
        embeds: ``(n_classes * n_per_class, dim)`` float32, L2-normalised.
        labels: ``(n_classes * n_per_class,)`` int64.
    """
    assert n_classes <= dim, "n_classes must be ≤ dim to guarantee orthogonal centroids"
    torch.manual_seed(seed)
    # Orthogonal unit vectors as class centroids
    centroids = F.normalize(torch.eye(n_classes, dim), dim=-1)          # (C, D)
    emb = centroids.unsqueeze(1).expand(-1, n_per_class, -1).clone()    # (C, N, D)
    emb = emb + noise * torch.randn_like(emb)
    emb = F.normalize(emb.reshape(-1, dim), dim=-1)                     # (C*N, D)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)     # (C*N,)
    return emb, labels


def _make_drift_data(n_classes=_N_CLASSES, dim=_DIM_IN, n_per_class=_N_DRIFT, drift=0.4, seed=1):
    """Synthetic drifted embeddings: larger deviation from class centroids.

    The drift magnitude is chosen to be large enough that most drifted samples
    fall below the default vigilance threshold (0.85), forcing new slot
    allocation rather than overwriting existing clean slots.

    Requires ``n_classes <= dim`` (same constraint as :func:`_make_clean_data`).

    Returns:
        embeds: ``(n_classes * n_per_class, dim)`` float32, L2-normalised.
        labels: ``(n_classes * n_per_class,)`` int64.
    """
    assert n_classes <= dim, "n_classes must be ≤ dim to guarantee orthogonal centroids"
    torch.manual_seed(seed)
    centroids = F.normalize(torch.eye(n_classes, dim), dim=-1)
    emb = centroids.unsqueeze(1).expand(-1, n_per_class, -1).clone()
    emb = emb + drift * torch.randn_like(emb)
    emb = F.normalize(emb.reshape(-1, dim), dim=-1)
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)
    return emb, labels


def _evaluate(fam, embeds, labels, batch_size=64):
    """Return top-1 accuracy (%) of ``fam`` on ``(embeds, labels)``."""
    fam.eval()
    correct = 0
    total = len(labels)
    with torch.no_grad():
        for i in range(0, total, batch_size):
            x = embeds[i: i + batch_size]
            y = labels[i: i + batch_size]
            preds = fam(x).argmax(dim=1)
            correct += (preds == y).sum().item()
    return 100.0 * correct / total


def _run_drift_scenario(fam):
    """Execute the two-phase G10 drift scenario and return BWT.

    Phase 1 → learn clean embeddings, evaluate.
    Phase 2 → learn drifted embeddings, re-evaluate clean.
    BWT = acc_clean_after_drift − acc_clean_before_drift.
    """
    clean_emb, clean_lbl = _make_clean_data()
    drift_emb, drift_lbl = _make_drift_data()

    # Phase 1: learn clean domain
    with torch.no_grad():
        fam.learn_local(clean_emb, clean_lbl)
    acc_t1 = _evaluate(fam, clean_emb, clean_lbl)

    # Phase 2: learn drift domain
    with torch.no_grad():
        fam.learn_local(drift_emb, drift_lbl)
    acc_t2 = _evaluate(fam, clean_emb, clean_lbl)

    bwt = acc_t2 - acc_t1
    return bwt, acc_t1, acc_t2


# ---------------------------------------------------------------------------
# G10 Drift — baseline FAM (no full-stack options, sanity check)
# ---------------------------------------------------------------------------

class TestG10DriftBaseline:
    """Verify the drift scenario works as expected with vanilla FAM."""

    def test_bwt_above_threshold(self):
        """Vanilla FAM must achieve BWT > -5 % (sanity check)."""
        fam = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=1000, core_vigilance=0.7,
        )
        bwt, acc_t1, acc_t2 = _run_drift_scenario(fam)
        assert bwt > _BWT_THRESHOLD, (
            f"Baseline G10 FAIL: BWT={bwt:+.2f}%  "
            f"(T1={acc_t1:.2f}%, T2={acc_t2:.2f}%)"
        )


# ---------------------------------------------------------------------------
# G10 Drift — quantized storage only (bfloat16)
# ---------------------------------------------------------------------------

class TestG10DriftQuantized:
    """bfloat16 prototype storage must not break non-destructive slot allocation."""

    def test_bwt_above_threshold(self):
        fam = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=1000, core_vigilance=0.7,
            use_bfloat16=True,
        )
        bwt, acc_t1, acc_t2 = _run_drift_scenario(fam)
        assert bwt > _BWT_THRESHOLD, (
            f"Quantized G10 FAIL: BWT={bwt:+.2f}%  "
            f"(T1={acc_t1:.2f}%, T2={acc_t2:.2f}%)"
        )


# ---------------------------------------------------------------------------
# G10 Drift — MetricAdapter only
# ---------------------------------------------------------------------------

class TestG10DriftAdapter:
    """MetricAdapter projection must preserve non-destructive slot allocation."""

    @pytest.fixture
    def adapter(self):
        torch.manual_seed(42)
        return MetricAdapter(input_dim=_DIM_IN, output_dim=_DIM_OUT)

    def test_bwt_above_threshold(self, adapter):
        fam = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=1000, core_vigilance=0.7,
            adapter=adapter,
        )
        bwt, acc_t1, acc_t2 = _run_drift_scenario(fam)
        assert bwt > _BWT_THRESHOLD, (
            f"Adapter G10 FAIL: BWT={bwt:+.2f}%  "
            f"(T1={acc_t1:.2f}%, T2={acc_t2:.2f}%)"
        )


# ---------------------------------------------------------------------------
# G10 Drift — Full Stack: bfloat16 + Adapter + NSTP
# ---------------------------------------------------------------------------

class TestG10DriftFullStack:
    """Full-stack G10 Drift Benchmark (G20 Continual-Learning Stress).

    Enables all three production features simultaneously:
      * ``use_bfloat16=True``  — quantized storage
      * :class:`~adapter.MetricAdapter` — metric projection
      * :class:`~nstp.NSTPController`  — lateral inhibition during inference

    BWT must remain > -5 % on the G10 Drift Benchmark to avoid triggering
    the G20 kill condition.
    """

    @pytest.fixture
    def full_stack_fam(self):
        torch.manual_seed(42)
        adapter = MetricAdapter(input_dim=_DIM_IN, output_dim=_DIM_OUT)
        nstp = NSTPController(sibling_threshold=0.85, depth_epsilon=0.10)
        return FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=1000, core_vigilance=0.7,
            use_bfloat16=True,
            adapter=adapter,
            nstp=nstp,
        )

    def test_bwt_above_kill_condition(self, full_stack_fam):
        """BWT > -5 % with full stack — G20 kill condition must not fire."""
        bwt, acc_t1, acc_t2 = _run_drift_scenario(full_stack_fam)
        assert bwt > _BWT_THRESHOLD, (
            f"G20 KILL CONDITION MET — Full-stack G10 BWT={bwt:+.2f}% ≤ -5%  "
            f"(T1={acc_t1:.2f}%, T2={acc_t2:.2f}%)"
        )

    def test_nstp_stored_on_fam(self, full_stack_fam):
        """NSTPController must be accessible via fam.nstp."""
        assert isinstance(full_stack_fam.nstp, NSTPController)

    def test_adapter_stored_on_fam(self, full_stack_fam):
        """MetricAdapter must be accessible via fam.adapter."""
        assert isinstance(full_stack_fam.adapter, MetricAdapter)

    def test_cam_uses_bfloat16(self, full_stack_fam):
        """Core CAM keys must be stored in bfloat16 when quantized storage is on."""
        assert full_stack_fam.core_cam.keys.dtype == torch.bfloat16

    def test_cam_key_dim_matches_adapter_output(self, full_stack_fam):
        """The CAM key dimension must equal adapter.output_dim."""
        assert full_stack_fam.core_cam.key_dim == full_stack_fam.adapter.output_dim

    def test_forward_output_shape(self, full_stack_fam):
        """forward() must return (B, value_dim) regardless of full-stack config."""
        x = torch.randn(8, _DIM_IN)
        # Populate memory first so forward returns meaningful output
        lbl = torch.randint(0, _N_CLASSES, (8,))
        full_stack_fam.learn_local(x, lbl)
        out = full_stack_fam(x)
        assert out.shape == (8, _N_CLASSES)

    def test_nstp_does_not_affect_learning(self, full_stack_fam):
        """NSTP is inference-only; slot counts must match a non-NSTP FAM."""
        torch.manual_seed(42)
        adapter_no_nstp = MetricAdapter(input_dim=_DIM_IN, output_dim=_DIM_OUT)
        fam_no_nstp = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=1000, core_vigilance=0.7,
            use_bfloat16=True,
            adapter=adapter_no_nstp,
            nstp=None,
        )

        clean_emb, clean_lbl = _make_clean_data(seed=99)
        with torch.no_grad():
            full_stack_fam.learn_local(clean_emb, clean_lbl)
            fam_no_nstp.learn_local(clean_emb, clean_lbl)

        # Slot counts must be identical — NSTP must not alter learn_local
        n_full = full_stack_fam.core_cam.occupied.sum().item()
        n_base = fam_no_nstp.core_cam.occupied.sum().item()
        assert n_full == n_base, (
            f"NSTP unexpectedly changed slot count: full_stack={n_full}, base={n_base}"
        )


# ---------------------------------------------------------------------------
# G10 Drift — NSTP integration smoke test
# ---------------------------------------------------------------------------

class TestNSTPIntegration:
    """NSTP is wired into ContinuousCAM.forward() via the nstp= parameter."""

    def test_nstp_forward_does_not_crash(self):
        """forward() with nstp=NSTPController() must not raise."""
        nstp = NSTPController()
        fam = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=256, core_vigilance=0.7,
            nstp=nstp,
        )
        x = torch.randn(4, _DIM_IN)
        lbl = torch.randint(0, _N_CLASSES, (4,))
        fam.learn_local(x, lbl)
        out = fam(x)
        assert out.shape == (4, _N_CLASSES)

    def test_nstp_output_finite(self):
        """All outputs from NSTP-enabled forward() must be finite."""
        nstp = NSTPController(sibling_threshold=0.80)
        fam = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=256, core_vigilance=0.7,
            nstp=nstp,
        )
        clean_emb, clean_lbl = _make_clean_data(n_per_class=10)
        fam.learn_local(clean_emb, clean_lbl)
        out = fam(clean_emb)
        assert torch.isfinite(out).all(), "NSTP-enabled forward produced non-finite outputs"

    def test_nstp_none_and_nstp_enabled_same_occupancy(self):
        """Enabling NSTP must not change the number of occupied slots."""
        clean_emb, clean_lbl = _make_clean_data(n_per_class=20, seed=7)

        torch.manual_seed(0)
        fam_plain = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=512, core_vigilance=0.7,
            nstp=None,
        )
        torch.manual_seed(0)
        fam_nstp = FastAssociativeMemory(
            input_dim=_DIM_IN, value_dim=_N_CLASSES,
            core_entries=512, core_vigilance=0.7,
            nstp=NSTPController(),
        )

        with torch.no_grad():
            fam_plain.learn_local(clean_emb, clean_lbl)
            fam_nstp.learn_local(clean_emb, clean_lbl)

        assert (fam_plain.core_cam.occupied.sum().item() ==
                fam_nstp.core_cam.occupied.sum().item())
