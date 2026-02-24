"""
tests/test_benchmark_nstp_latency.py — Unit tests for the G15 NSTP latency benchmark.

Tests the four NSTP stage functions for:
  - Correct similarity matrix computation
  - Root election selects the nearest node to the query
  - Path cost BFS sets root cost to 0 and all others non-negative
  - Sibling link / blocking port identification
  - _percentile helper
  - End-to-end _run_benchmark returns well-formed timing structures
  - Pass/fail criterion against the 1ms kill threshold
"""

import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Allow import from repo root regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_nstp_latency import (
    _stage_similarity_matrix,
    _stage_root_election,
    _stage_path_cost,
    _stage_sibling_links,
    _percentile,
    _run_benchmark,
    _KILL_THRESHOLD_US,
)

torch.manual_seed(0)

K = 10    # small subgraph for fast unit tests
DIM = 32  # embedding dimension


# ---------------------------------------------------------------------------
# Stage 1 — Similarity Matrix
# ---------------------------------------------------------------------------

class TestStageSimilarityMatrix:
    def test_shape(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        assert sim.shape == (K, K)

    def test_diagonal_is_one(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        assert torch.allclose(sim.diagonal(), torch.ones(K), atol=1e-5)

    def test_symmetric(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        assert torch.allclose(sim, sim.T, atol=1e-5)

    def test_values_in_range(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        assert sim.min() >= -1.0 - 1e-5
        assert sim.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Stage 2 — Root Election
# ---------------------------------------------------------------------------

class TestStageRootElection:
    def test_returns_int(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        query = F.normalize(torch.randn(DIM), dim=0)
        root = _stage_root_election(nodes, query)
        assert isinstance(root, int)

    def test_valid_index(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        query = F.normalize(torch.randn(DIM), dim=0)
        root = _stage_root_election(nodes, query)
        assert 0 <= root < K

    def test_selects_nearest_node(self):
        # Construct nodes where the second one (index 1) is identical to the query.
        query = F.normalize(torch.randn(DIM), dim=0)
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        nodes[1] = query.clone()
        root = _stage_root_election(nodes, query)
        assert root == 1


# ---------------------------------------------------------------------------
# Stage 3 — Path Cost
# ---------------------------------------------------------------------------

class TestStagePathCost:
    def _make_sim(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        return _stage_similarity_matrix(nodes)

    def test_root_cost_is_zero(self):
        sim = self._make_sim()
        for root in [0, K // 2, K - 1]:
            path_cost = _stage_path_cost(sim, root)
            assert path_cost[root] == 0.0, f"path_cost[root={root}] should be 0"

    def test_all_nodes_reachable(self):
        sim = self._make_sim()
        path_cost = _stage_path_cost(sim, root=0)
        assert len(path_cost) == K
        assert all(c < float("inf") for c in path_cost)

    def test_all_costs_non_negative(self):
        sim = self._make_sim()
        path_cost = _stage_path_cost(sim, root=0)
        assert all(c >= 0.0 for c in path_cost)

    def test_length_equals_k(self):
        sim = self._make_sim()
        path_cost = _stage_path_cost(sim, root=0)
        assert len(path_cost) == K


# ---------------------------------------------------------------------------
# Stage 4 — Sibling Links & Blocking Ports
# ---------------------------------------------------------------------------

class TestStageSiblingLinks:
    def test_returns_list_of_tuples(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        path_cost = _stage_path_cost(sim, root=0)
        blocking = _stage_sibling_links(sim, root=0, path_cost=path_cost)
        assert isinstance(blocking, list)
        for item in blocking:
            assert isinstance(item, tuple) and len(item) == 2

    def test_root_not_in_blocking_ports(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        path_cost = _stage_path_cost(sim, root=0)
        blocking = _stage_sibling_links(sim, root=0, path_cost=path_cost)
        for u, v in blocking:
            assert u != 0 and v != 0, "Root node should not appear in blocking ports"

    def test_no_self_loops(self):
        nodes = F.normalize(torch.randn(K, DIM), dim=-1)
        sim = _stage_similarity_matrix(nodes)
        path_cost = _stage_path_cost(sim, root=0)
        blocking = _stage_sibling_links(sim, root=0, path_cost=path_cost)
        for u, v in blocking:
            assert u != v


# ---------------------------------------------------------------------------
# _percentile helper
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_median_of_sorted_list(self):
        vals = list(range(1, 100))  # 1..99
        median = _percentile(vals, 50)
        assert abs(median - 50.0) < 1.0

    def test_p0_is_min(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert _percentile(vals, 0) == pytest.approx(1.0, abs=0.1)

    def test_p100_is_max(self):
        vals = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert _percentile(vals, 100) == pytest.approx(5.0, abs=0.1)

    def test_empty_list_returns_zero(self):
        assert _percentile([], 99) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 99) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# _run_benchmark — integration / smoke test
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    def test_returns_correct_structure(self):
        stage_timings, total_timings = _run_benchmark(k=5, trials=10, seed=0, dim=16)
        assert set(stage_timings.keys()) == {
            "similarity_matrix", "root_election", "path_cost", "sibling_links"
        }
        assert len(total_timings) == 10
        for key, vals in stage_timings.items():
            assert len(vals) == 10, f"Expected 10 timings for {key}"

    def test_all_timings_positive(self):
        stage_timings, total_timings = _run_benchmark(k=5, trials=10, seed=1, dim=16)
        for key, vals in stage_timings.items():
            assert all(v >= 0.0 for v in vals), f"Negative timing in {key}"
        assert all(v >= 0.0 for v in total_timings)

    def test_total_at_least_sum_of_stages(self):
        """Total wall time must be >= sum of individual stage times for each trial."""
        stage_timings, total_timings = _run_benchmark(k=5, trials=20, seed=2, dim=16)
        for i in range(20):
            stage_sum = sum(stage_timings[stage_name][i] for stage_name in stage_timings)
            assert total_timings[i] >= stage_sum - 1.0, (
                f"Trial {i}: total {total_timings[i]:.3f} µs < stage sum "
                f"{stage_sum:.3f} µs (more than 1 µs tolerance)"
            )

    def test_benchmark_passes_kill_threshold(self):
        """G15 success criterion: p99 total NSTP latency < 1ms on CPU."""
        _, total_timings = _run_benchmark(k=50, trials=1000, seed=42, dim=128)
        p99 = _percentile(total_timings, 99)
        assert p99 < _KILL_THRESHOLD_US, (
            f"G15 FAIL: p99 = {p99:.1f} µs ≥ {_KILL_THRESHOLD_US:.0f} µs kill threshold"
        )
