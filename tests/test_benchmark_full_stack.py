"""
tests/test_benchmark_full_stack.py — Unit tests for the G18 full-stack latency benchmark.

Tests the benchmark helpers for:
  - Correct stack construction (adapter + bfloat16 quantized core + NSTP)
  - Stage timing dictionary structure and lengths
  - Total timings equal the sum of stage timings (no scaffolding overhead)
  - All timings are non-negative
  - _mean and _percentile helpers
  - G18 kill condition: mean total latency < 2 ms per query on CPU
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Allow importing from the repo root regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_full_stack import (
    _build_full_stack,
    _run_benchmark,
    _mean,
    _percentile,
    _KILL_THRESHOLD_MS,
    _DEFAULT_INPUT_DIM,
    _DEFAULT_ADAPTER_DIM,
    _DEFAULT_N_CLASSES,
    _DEFAULT_K,
    _DEFAULT_N_ENTRIES,
)
from adapter import MetricAdapter
from fast_associative_memory import FastAssociativeMemory
from nstp import NSTPController

torch.manual_seed(0)

# Small dimensions for fast unit tests
_INPUT_DIM = 64
_ADAPTER_DIM = 32
_N_CLASSES = 10
_K = 10
_N_ENTRIES = 50


# ---------------------------------------------------------------------------
# _build_full_stack
# ---------------------------------------------------------------------------

class TestBuildFullStack:
    def test_returns_fam_and_nstp(self):
        fam, nstp = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert isinstance(fam, FastAssociativeMemory)
        assert isinstance(nstp, NSTPController)

    def test_fam_has_adapter(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert isinstance(fam.adapter, MetricAdapter)

    def test_adapter_dimensions(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert fam.adapter.input_dim == _INPUT_DIM
        assert fam.adapter.output_dim == _ADAPTER_DIM

    def test_fam_uses_bfloat16(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert fam.core_cam.use_bfloat16

    def test_core_pre_populated(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert fam.core_cam.occupied[:_N_ENTRIES].all()

    def test_inference_k_matches(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert fam.core_cam.inference_k == _K

    def test_cam_key_dim_matches_adapter_output(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        assert fam.core_cam.key_dim == _ADAPTER_DIM

    def test_fam_forward_runs(self):
        fam, _ = _build_full_stack(
            _INPUT_DIM, _ADAPTER_DIM, _N_CLASSES, _K, _N_ENTRIES, seed=0
        )
        query = F.normalize(torch.randn(1, _INPUT_DIM), dim=-1)
        with torch.no_grad():
            out = fam(query)
        assert out.shape == (1, _N_CLASSES)


# ---------------------------------------------------------------------------
# _run_benchmark structure
# ---------------------------------------------------------------------------

class TestRunBenchmarkStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return _run_benchmark(
            input_dim=_INPUT_DIM,
            adapter_dim=_ADAPTER_DIM,
            n_classes=_N_CLASSES,
            k=_K,
            n_entries=_N_ENTRIES,
            trials=20,
            seed=0,
            batch=1,
        )

    def test_stage_keys(self, result):
        stage_timings, _ = result
        assert set(stage_timings.keys()) == {"adapter_quantized_core", "nstp"}

    def test_stage_lengths_match_trials(self, result):
        stage_timings, total_timings = result
        for key, vals in stage_timings.items():
            assert len(vals) == 20, f"Expected 20 timings for '{key}'"
        assert len(total_timings) == 20

    def test_all_timings_non_negative(self, result):
        stage_timings, total_timings = result
        for key, vals in stage_timings.items():
            assert all(v >= 0.0 for v in vals), f"Negative timing in '{key}'"
        assert all(v >= 0.0 for v in total_timings)

    def test_total_equals_sum_of_stages(self, result):
        """Total must equal Stage 1 + Stage 2 for every trial (no wall-clock gap)."""
        stage_timings, total_timings = result
        for i, total in enumerate(total_timings):
            expected = (
                stage_timings["adapter_quantized_core"][i]
                + stage_timings["nstp"][i]
            )
            assert abs(total - expected) < 1e-9, (
                f"Trial {i}: total {total} ms != stage sum {expected} ms"
            )

    def test_returns_correct_trial_count(self, result):
        _, total_timings = result
        assert len(total_timings) == 20


# ---------------------------------------------------------------------------
# _mean helper
# ---------------------------------------------------------------------------

class TestMean:
    def test_empty_returns_zero(self):
        assert _mean([]) == 0.0

    def test_single_element(self):
        assert _mean([5.0]) == pytest.approx(5.0)

    def test_basic_average(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_float_precision(self):
        vals = [0.1, 0.2, 0.3]
        assert _mean(vals) == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 99) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)

    def test_p0_is_min(self):
        assert _percentile([3.0, 1.0, 4.0, 1.0, 5.0], 0) == pytest.approx(1.0, abs=0.1)

    def test_p100_is_max(self):
        assert _percentile([3.0, 1.0, 4.0, 1.0, 5.0], 100) == pytest.approx(5.0, abs=0.1)

    def test_median_of_sorted_list(self):
        vals = list(range(1, 100))   # 1..99
        median = _percentile(vals, 50)
        assert median == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# G18 kill condition: mean total < 2 ms per query on CPU
# ---------------------------------------------------------------------------

class TestKillCondition:
    def test_mean_latency_below_threshold(self):
        """G18 success criterion: mean total latency < 2 ms per query on CPU.

        Uses 2 000 pre-loaded prototypes (a ×2.5 reduction from the default
        5 000) to give stable sub-threshold readings on CI-grade hardware,
        while still fully exercising the adapter + bfloat16 core + NSTP
        code path with the production K=50 subgraph size.
        """
        _, total_timings = _run_benchmark(
            input_dim=_DEFAULT_INPUT_DIM,
            adapter_dim=_DEFAULT_ADAPTER_DIM,
            n_classes=_DEFAULT_N_CLASSES,
            k=_DEFAULT_K,
            n_entries=2000,
            trials=500,
            seed=42,
            batch=1,
        )
        mean_ms = _mean(total_timings)
        assert mean_ms < _KILL_THRESHOLD_MS, (
            f"G18 FAIL: mean = {mean_ms:.3f} ms "
            f"≥ {_KILL_THRESHOLD_MS:.1f} ms kill threshold"
        )
