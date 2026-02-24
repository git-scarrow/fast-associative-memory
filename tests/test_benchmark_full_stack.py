"""
tests/test_benchmark_full_stack.py — Unit tests for the G18 full-stack latency benchmark.

Tests cover:
  - _build_stack() returns the correct module types
  - _run_benchmark() returns well-formed timing structures
  - All per-trial timings are non-negative
  - Per-trial total time >= sum of individual stage times (within tolerance)
  - _mean() and _percentile() helpers behave correctly
  - _print_report() returns the correct bool based on the kill threshold
  - The benchmark passes with a small, fast configuration
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import MetricAdapter
from associative_core import ContinuousCAM
from nstp import NSTPController
from benchmarks.benchmark_full_stack import (
    _build_stack,
    _mean,
    _percentile,
    _print_report,
    _run_benchmark,
    _KILL_THRESHOLD_MS,
)

torch.manual_seed(0)

# Small configuration for fast unit tests
_DIM_IN = 64
_DIM_PROJ = 32
_VALUE_DIM = 10
_CAM_ENTRIES = 100
_K = 10
_TRIALS = 20
_WARMUP = 5
_SEED = 0


# ---------------------------------------------------------------------------
# _build_stack
# ---------------------------------------------------------------------------

class TestBuildStack:
    def test_returns_three_modules(self):
        result = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert len(result) == 3

    def test_adapter_type(self):
        adapter, _, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert isinstance(adapter, MetricAdapter)

    def test_cam_type(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert isinstance(cam, ContinuousCAM)

    def test_nstp_type(self):
        _, _, nstp = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert isinstance(nstp, NSTPController)

    def test_adapter_dimensions(self):
        adapter, _, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert adapter.input_dim == _DIM_IN
        assert adapter.output_dim == _DIM_PROJ

    def test_cam_bfloat16_keys(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert cam.keys.dtype == torch.bfloat16

    def test_cam_bfloat16_values(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert cam.values.dtype == torch.bfloat16

    def test_cam_inference_k(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert cam.inference_k == _K

    def test_cam_fully_occupied(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        assert cam.occupied.all()

    def test_cam_keys_unit_norm(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        norms = cam.keys.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones(_CAM_ENTRIES), atol=1e-2)

    def test_adapter_forward_shape(self):
        adapter, _, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        x = torch.randn(1, _DIM_IN)
        with torch.no_grad():
            out = adapter(x)
        assert out.shape == (1, _DIM_PROJ)

    def test_cam_forward_shape(self):
        _, cam, _ = _build_stack(_DIM_IN, _DIM_PROJ, _VALUE_DIM, _CAM_ENTRIES, _K, _SEED)
        x = F.normalize(torch.randn(1, _DIM_PROJ), dim=-1)
        with torch.no_grad():
            out = cam(x)
        assert out.shape == (1, _VALUE_DIM)


# ---------------------------------------------------------------------------
# _mean and _percentile helpers
# ---------------------------------------------------------------------------

class TestMean:
    def test_empty_returns_zero(self):
        assert _mean([]) == 0.0

    def test_single_value(self):
        assert _mean([5.0]) == pytest.approx(5.0)

    def test_uniform_values(self):
        assert _mean([2.0, 2.0, 2.0]) == pytest.approx(2.0)

    def test_mixed_values(self):
        assert _mean([1.0, 3.0]) == pytest.approx(2.0)


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 99) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)

    def test_p0_is_min(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert _percentile(vals, 0) == pytest.approx(1.0, abs=0.1)

    def test_p100_is_max(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert _percentile(vals, 100) == pytest.approx(5.0, abs=0.1)

    def test_median_of_sorted_list(self):
        vals = list(range(1, 100))
        median = _percentile(vals, 50)
        assert abs(median - 50.0) < 1.0


# ---------------------------------------------------------------------------
# _run_benchmark — structure and correctness
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def _run(self):
        return _run_benchmark(
            dim_in=_DIM_IN,
            dim_proj=_DIM_PROJ,
            value_dim=_VALUE_DIM,
            cam_entries=_CAM_ENTRIES,
            k=_K,
            trials=_TRIALS,
            warmup=_WARMUP,
            seed=_SEED,
        )

    def test_returns_two_items(self):
        result = self._run()
        assert len(result) == 2

    def test_stage_keys(self):
        stage_timings, _ = self._run()
        assert set(stage_timings.keys()) == {"adapter", "quantized_core", "nstp"}

    def test_trial_count(self):
        stage_timings, total_timings = self._run()
        assert len(total_timings) == _TRIALS
        for key, vals in stage_timings.items():
            assert len(vals) == _TRIALS, f"Expected {_TRIALS} timings for stage '{key}'"

    def test_all_stage_timings_non_negative(self):
        stage_timings, _ = self._run()
        for key, vals in stage_timings.items():
            assert all(v >= 0.0 for v in vals), f"Negative timing in stage '{key}'"

    def test_all_total_timings_non_negative(self):
        _, total_timings = self._run()
        assert all(v >= 0.0 for v in total_timings)

    def test_total_at_least_sum_of_stages(self):
        """Total wall time must be >= sum of stage times for each trial."""
        stage_timings, total_timings = self._run()
        tolerance_ms = 0.1  # allow 0.1 ms tolerance for perf_counter overhead
        for i in range(_TRIALS):
            stage_sum = sum(stage_timings[stage][i] for stage in stage_timings)
            assert total_timings[i] >= stage_sum - tolerance_ms, (
                f"Trial {i}: total {total_timings[i]:.4f} ms < "
                f"stage sum {stage_sum:.4f} ms (tolerance {tolerance_ms} ms)"
            )

    def test_warmup_not_in_output(self):
        """Warmup trials must not appear in the returned timing lists."""
        stage_timings, total_timings = self._run()
        assert len(total_timings) == _TRIALS

    def test_reproducible_with_same_seed(self):
        """Two runs with the same seed should produce means within 20% of each other."""
        _, t1 = self._run()
        _, t2 = self._run()
        mean1 = _mean(t1)
        mean2 = _mean(t2)
        assert mean1 > 0.0 and mean2 > 0.0
        assert abs(mean1 - mean2) / mean1 <= 0.20, (
            f"Runs not reproducible: mean1={mean1:.4f} ms, mean2={mean2:.4f} ms "
            f"(difference > 20%)"
        )


# ---------------------------------------------------------------------------
# _print_report — pass/fail logic
# ---------------------------------------------------------------------------

class TestPrintReport:
    def _make_timings(self, total_mean_ms: float, n: int = 10):
        """Create synthetic stage and total timing lists centered on total_mean_ms."""
        stage_timings = {
            "adapter":        [total_mean_ms * 0.05] * n,
            "quantized_core": [total_mean_ms * 0.80] * n,
            "nstp":           [total_mean_ms * 0.15] * n,
        }
        total_timings = [total_mean_ms] * n
        return stage_timings, total_timings

    def test_pass_when_below_threshold(self, capsys):
        stage_timings, total_timings = self._make_timings(1.0)
        passed = _print_report(stage_timings, total_timings, k=50, trials=10)
        assert passed is True
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_fail_when_above_threshold(self, capsys):
        stage_timings, total_timings = self._make_timings(3.0)
        passed = _print_report(stage_timings, total_timings, k=50, trials=10)
        assert passed is False
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_fail_at_exactly_threshold(self, capsys):
        """Exactly at threshold should fail (strict < comparison)."""
        stage_timings, total_timings = self._make_timings(_KILL_THRESHOLD_MS)
        passed = _print_report(stage_timings, total_timings, k=50, trials=10)
        assert passed is False

    def test_kill_threshold_constant(self):
        assert _KILL_THRESHOLD_MS == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# G18 end-to-end: small config guaranteed to pass the kill threshold
# ---------------------------------------------------------------------------

class TestKillThreshold:
    def test_small_config_passes(self):
        """A tiny stack (100 entries, K=10) must beat the 2 ms kill threshold."""
        stage_timings, total_timings = _run_benchmark(
            dim_in=64,
            dim_proj=32,
            value_dim=10,
            cam_entries=100,
            k=10,
            trials=100,
            warmup=10,
            seed=42,
        )
        mean_ms = _mean(total_timings)
        assert mean_ms < _KILL_THRESHOLD_MS, (
            f"G18 small-config FAIL: mean = {mean_ms:.3f} ms "
            f">= {_KILL_THRESHOLD_MS:.1f} ms kill threshold"
        )
